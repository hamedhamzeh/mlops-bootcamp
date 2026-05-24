import typer
from rich import print

from airbnb_ops.config import PipelineConfig
from airbnb_ops.extract import read_csv_checked
from airbnb_ops.pii import handle_pii
from airbnb_ops.transform import build_neighbourhood_summary
from airbnb_ops.validate import validate_summary

app = typer.Typer()


@app.command()
def run():
    config = PipelineConfig()

    print("[blue]Reading raw csv files...[/blue]")

    listings = read_csv_checked(config.listings_path)

    segments = read_csv_checked(config.segments_path)

    print("[blue]Handling PII...[/blue]")

    listings_clean = handle_pii(listings)

    print("[blue]Building neighbourhood summary...[/blue]")

    summary = build_neighbourhood_summary(
        listings_clean,
        segments
    )

    print("[blue]Validating output...[/blue]")

    validate_summary(summary)

    print("[blue]Writing processed CSV...[/blue]")

    summary.to_csv(
        config.output_path,
        index=False
    )

    report = f"""
    # HW01-A Run Report

    ## Pipeline Status
    SUCCESS

    ## Input Files
    - {config.listings_path}
    - {config.segments_path}

    ## Output File
    - {config.output_path}

    ## Summary Statistics
    - Number of neighbourhoods: {len(summary)}
    - Total listings: {summary['num_listings'].sum()}
    """

    config.report_path.write_text(report)

    print("[green]Pipeline completed successfully.[/green]")


if __name__ == "__main__":
    app()